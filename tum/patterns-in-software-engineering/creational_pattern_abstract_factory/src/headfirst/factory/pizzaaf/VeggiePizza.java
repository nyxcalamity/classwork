package headfirst.factory.pizzaaf;

public class VeggiePizza extends Pizza {
	PizzaToppingFactory ingredientFactory;
 
	public VeggiePizza(PizzaToppingFactory ingredientFactory) {
		this.ingredientFactory = ingredientFactory;
	}
 
	void prepare() {
		System.out.println("Preparing " + name);
		dough = ingredientFactory.createDough();
		sauce = ingredientFactory.createSauce();
		cheese = ingredientFactory.createCheese();
		veggies = ingredientFactory.createVeggies();
	}
}
