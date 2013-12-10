package headfirst.factory.pizzaaf;

public class ClamPizza extends Pizza {
	PizzaToppingFactory ingredientFactory;
 
	public ClamPizza(PizzaToppingFactory ingredientFactory) {
		this.ingredientFactory = ingredientFactory;
	}
 
	void prepare() {
		System.out.println("Preparing " + name);
		dough = ingredientFactory.createDough();
		sauce = ingredientFactory.createSauce();
		cheese = ingredientFactory.createCheese();
		clam = ingredientFactory.createClam();
	}
}
