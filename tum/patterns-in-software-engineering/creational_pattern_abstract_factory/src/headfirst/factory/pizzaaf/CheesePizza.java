package headfirst.factory.pizzaaf;

public class CheesePizza extends Pizza {
	PizzaToppingFactory ingredientFactory;
 
	public CheesePizza(PizzaToppingFactory ingredientFactory) {
		this.ingredientFactory = ingredientFactory;
	}
 
	void prepare() {
		System.out.println("Preparing " + name);
		dough = ingredientFactory.createDough();
		sauce = ingredientFactory.createSauce();
		cheese = ingredientFactory.createCheese();
	}
}
